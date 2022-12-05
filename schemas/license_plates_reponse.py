class CertificateContent_license_plates():

    def __init__(self, license_plate_number,
                 registration_date,
                 hsn,
                 tsn,
                 vin,
                 fuel_grade,
                 vehicle_type,
                 emission_code,
                 particulate_reduction_system,license_plates_c1,license_plates_c2,license_plates_c3,license_plates_j
                 ):
        self.license_plate_number = license_plate_number

        self.registration_date = registration_date
        self.hsn = hsn
        self.tsn = tsn
        self.vin = vin
        self.fuel_grade = fuel_grade
        self.vehicle_type = vehicle_type
        self.emission_code = emission_code
        self.particulate_reduction_system = particulate_reduction_system 
        self.name = license_plates_c1
        self.first_name = license_plates_c2
        self.address = license_plates_c3
        self.vehicle_class = license_plates_j

    def isEmpty(self):
        clp_dict = self.toDict()
        con_str = ""
        for k, v in clp_dict.items():
            if k == "address":
                continue
            con_str += v
        return con_str.strip() == ''
    
    def _post_process_address(self, address):
        try:
            if "," in address:
                street_number, postcode_city = address.split(",")[:2]

                postcode_city_arr = postcode_city.strip().split(" ", 1)
                if len(postcode_city_arr)==2:
                    postcode, city =  postcode_city_arr
                else:
                    postcode, city = "", postcode_city
                
                import re
                match = re.search("\d", street_number)
                if match:
                    index_of_first_digit = match.start()
                    street, house_number = street_number[:index_of_first_digit], street_number[index_of_first_digit:]
                else:
                    street, house_number = street_number, ""
                return {
                    "street": street,
                    'house_number': house_number,
                    'postcode': postcode,
                    'city': city
                }
        except:
            pass
        return {"street": address}


    
    def toDict(self):
        address_dict = self._post_process_address(self.address.get('content',''))
        return {"vin": self.vin.get('content',''),
            "registration_date": self.registration_date.get('content','') ,
            "license_plate_number" : self.license_plate_number.get('content',''),
            "vehicle_type": self.vehicle_type.get('content',''),
            "hsn": self.hsn.get('content',''),
            "tsn" : self.tsn.get('content',''),
            "fuel_grade" : self.fuel_grade.get('content',''),
            "emission_code": self.emission_code.get('content',''),
            "particulate_reduction_system": self.particulate_reduction_system.get('content','').replace(" ", "").replace("0","O"),
            "name": self.name.get('content',''),
            "first_name": self.first_name.get('content',''),
            "address": address_dict,
            "vehicle_class": self.vehicle_class.get('content','')}



