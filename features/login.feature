# Feature: Coordinate-Based Form Automation using OCR

#   Scenario: Successful login using coordinates detected by OCR
#     Given I navigate to "https://tutorialsninja.com/demo/index.php?route=account/login" and perform OCR
#     # Then I see the page title as "My Account"
#     When I enter "suryaiiit.517@gmail.com" using label "E-Mail Address"
#     And I enter "password123" using label "Password"
#     And I click using label "Login"

# Feature: Coordinate-Based Form Automation using OCR

#   Scenario: Successful login using coordinates detected by OCR
#     Given I navigate to "https://demowebshop.tricentis.com/login" and perform OCR
#     # Then I see the page title as "My Account"
#     When I enter "gaddala.surya@dragonflytest.com" using label "Email"
#     And I enter "Test@123" using label "Password"
#     And I click using label "Login"

Feature: Coordinate-Based Form Automation using OCR

  Scenario: Successful login using coordinates detected by OCR
    Given I navigate to "https://hab.instarresearch.com/wix/56789/p159939540878.aspx?QClink=1" and perform OCR
    # Then I see the page title as "My Account"
    When I enter "1" using label "BG_Target"
    And I enter "1" using label "BG_DSE_Segment_Code"
    And I click using label "Next"
    And I wait for 10 seconds
    And I see the label "QID: TargetH"
    # And I click using label "Next"
    # And I click using label "Next"
    # # And I click using label "1_HAB"
    # # And I click using label "Next"
    # # And I click using label "United Arab Emirates"